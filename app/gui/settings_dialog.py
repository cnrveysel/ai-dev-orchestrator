"""Settings dialog. The worker executable is validated on save: the normal
`codex` launcher is refused and there is never a fallback."""
from __future__ import annotations

import copy

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from app.config.discovery import (
    ExecutableRejected, discover_claude, discover_codebase_memory, discover_codex_ninfer,
    validate_claude_executable, validate_worker_executable,
)
from app.config.settings import Settings
from app.gui.worker_profiles_tab import WorkerProfilesTab
from app.workers.profile_registry import WorkerProfileRegistry
from app.workers.profiles import LEGACY_PROFILE_ID

SANDBOX_MODES = ("workspace-write", "read-only", "danger-full-access")
UNSAFE_ARG_MARKERS = ("dangerously", "bypass", "--yolo", "skip-permissions")


def _path_row(edit: QLineEdit, title: str, candidates_fn=None) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(edit, 1)
    b = QPushButton("Browse…")

    def browse() -> None:
        p, _ = QFileDialog.getOpenFileName(w, title, edit.text(), "Executables (*.exe);;All files (*)")
        if p:
            edit.setText(p.replace("/", "\\"))
    b.clicked.connect(browse)
    h.addWidget(b)
    if candidates_fn:
        d = QPushButton("Detect")

        def detect() -> None:
            cands = candidates_fn()
            if not cands:
                QMessageBox.information(w, "Detect", "No candidates found. Use Browse to select the executable.")
                return
            from PySide6.QtWidgets import QInputDialog
            import time as _t
            items = [f"{c.path}    ({c.source}, modified {_t.strftime('%Y-%m-%d %H:%M', _t.localtime(c.mtime))})"
                     for c in cands]
            item, ok = QInputDialog.getItem(w, "Select executable", "Candidates (newest first):", items, 0, False)
            if ok:
                edit.setText(cands[items.index(item)].path)
        d.clicked.connect(detect)
        h.addWidget(d)
    return w


def _args_edit(args: list[str]) -> QPlainTextEdit:
    e = QPlainTextEdit("\n".join(args))
    e.setPlaceholderText("one argument per line")
    e.setMaximumHeight(60)
    return e


def _parse_args(edit: QPlainTextEdit) -> list[str]:
    return [line.strip() for line in edit.toPlainText().splitlines() if line.strip()]


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None, in_use_provider=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(760, 560)
        self.settings = copy.deepcopy(settings)
        self.full_test_profile: str | None = None  # V5: "Full test…" asked to save and test this profile
        s = self.settings
        tabs = QTabWidget()

        # ---------------- Claude
        t = QWidget()
        f = QFormLayout(t)
        self.c_exe = QLineEdit(s.claude.executable)
        f.addRow("Claude Code CLI:", _path_row(self.c_exe, "Select claude.exe", discover_claude))
        self.c_model = QComboBox()
        self.c_model.setEditable(True)
        self.c_model.addItems(["opus", "fable", "sonnet", "claude-opus-5-5", ""])
        self.c_model.setCurrentText(s.claude.model)
        f.addRow("Planner model (--model):", self.c_model)
        self.c_review_model = QComboBox()
        self.c_review_model.setEditable(True)
        self.c_review_model.addItems(["claude-opus-5-5", "opus", "sonnet", "fable", ""])
        self.c_review_model.setCurrentText(s.claude.review_model)
        f.addRow("Reviewer model (empty = planner model):", self.c_review_model)
        self.c_effort = QComboBox()
        self.c_effort.addItems(["", "low", "medium", "high", "xhigh", "max"])
        self.c_effort.setCurrentText(s.claude.effort)
        f.addRow("Effort (--effort, empty = default):", self.c_effort)
        c = s.claude

        def spin(value: int, lo: int, hi: int) -> QSpinBox:
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setValue(value)
            return w

        def money(value: float) -> QDoubleSpinBox:
            w = QDoubleSpinBox()
            w.setRange(0.0, 5.0)
            w.setSingleStep(0.05)
            w.setDecimals(2)
            w.setPrefix("$")
            w.setSpecialValueText("not applied")
            w.setValue(value)
            return w

        def pair(a: QWidget, b: QWidget) -> QWidget:
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel("plan"))
            h.addWidget(a)
            h.addWidget(QLabel("review"))
            h.addWidget(b)
            h.addStretch(1)
            return w

        self.c_plan_timeout = spin(c.plan_timeout_seconds, 10, 3600)
        self.c_review_timeout = spin(c.review_timeout_seconds, 10, 3600)
        self.c_smoke_timeout = spin(c.smoke_timeout_seconds, 10, 600)
        f.addRow("Hard timeout (s):", pair(self.c_plan_timeout, self.c_review_timeout))
        f.addRow("Smoke test timeout (s):", self.c_smoke_timeout)
        self.c_plan_turns = spin(c.plan_max_turns, 1, 50)
        self.c_review_turns = spin(c.review_max_turns, 1, 50)
        f.addRow("Max turns:", pair(self.c_plan_turns, self.c_review_turns))
        self.c_plan_tools = spin(c.plan_max_tool_calls, 0, 100)
        f.addRow("Max tool calls (planner; the reviewer has no tools):", self.c_plan_tools)
        from app.orchestration.reviewer import REVIEW_PROMPT_BYTES_CEILING
        self.c_review_prompt_bytes = spin(c.review_max_prompt_bytes, 2000, REVIEW_PROMPT_BYTES_CEILING)
        self.c_review_diff_bytes = spin(c.review_max_diff_bytes, 0, REVIEW_PROMPT_BYTES_CEILING)
        self.c_review_files = spin(c.review_max_changed_files, 1, 200)
        limits_row = QWidget()
        lr = QHBoxLayout(limits_row)
        lr.setContentsMargins(0, 0, 0, 0)
        for label, w in (("prompt bytes", self.c_review_prompt_bytes), ("diff bytes", self.c_review_diff_bytes),
                         ("files", self.c_review_files)):
            lr.addWidget(QLabel(label))
            lr.addWidget(w)
        lr.addStretch(1)
        f.addRow("Reviewer context hard limits:", limits_row)
        self.c_ck_timeout = spin(c.checkpoint_timeout_seconds, 10, 3600)
        self.c_ck_budget = money(c.checkpoint_max_budget_usd)
        self.c_ck_prompt_bytes = spin(c.checkpoint_max_prompt_bytes, 4000, REVIEW_PROMPT_BYTES_CEILING)
        self.c_ck_diff_bytes = spin(c.checkpoint_max_diff_bytes, 0, REVIEW_PROMPT_BYTES_CEILING)
        self.c_ck_files = spin(c.checkpoint_max_changed_files, 1, 300)
        ck_row = QWidget()
        cr = QHBoxLayout(ck_row)
        cr.setContentsMargins(0, 0, 0, 0)
        for label, w in (("timeout s", self.c_ck_timeout), ("budget", self.c_ck_budget),
                         ("prompt bytes", self.c_ck_prompt_bytes), ("diff bytes", self.c_ck_diff_bytes),
                         ("files", self.c_ck_files)):
            cr.addWidget(QLabel(label))
            cr.addWidget(w)
        cr.addStretch(1)
        f.addRow("V2 phase checkpoint:", ck_row)
        self.c_hf_timeout = spin(c.human_followup_timeout_seconds, 10, 3600)
        self.c_hf_budget = money(c.human_followup_max_budget_usd)
        self.c_hf_prompt_bytes = spin(c.human_followup_max_prompt_bytes, 6000, REVIEW_PROMPT_BYTES_CEILING)
        hf_row = QWidget()
        hr = QHBoxLayout(hf_row)
        hr.setContentsMargins(0, 0, 0, 0)
        for label, w in (("timeout s", self.c_hf_timeout), ("budget", self.c_hf_budget),
                         ("prompt bytes", self.c_hf_prompt_bytes)):
            hr.addWidget(QLabel(label))
            hr.addWidget(w)
        hr.addStretch(1)
        f.addRow("V4 human follow-up:", hf_row)
        self.c_ep_timeout = spin(c.evidence_plan_timeout_seconds, 10, 3600)
        self.c_ep_budget = money(c.evidence_plan_max_budget_usd)
        self.c_ep_prompt_bytes = spin(c.evidence_plan_max_prompt_bytes, 6000, REVIEW_PROMPT_BYTES_CEILING)
        ep_row = QWidget()
        er = QHBoxLayout(ep_row)
        er.setContentsMargins(0, 0, 0, 0)
        for label, w in (("timeout s", self.c_ep_timeout), ("budget", self.c_ep_budget),
                         ("prompt bytes", self.c_ep_prompt_bytes)):
            er.addWidget(QLabel(label))
            er.addWidget(w)
        er.addStretch(1)
        f.addRow("V4 evidence plan:", ep_row)
        self.c_cr_timeout = spin(c.consistency_timeout_seconds, 10, 3600)
        self.c_cr_turns = spin(c.consistency_max_turns, 1, 10)
        self.c_cr_budget = money(c.consistency_max_budget_usd)
        self.c_cr_prompt_bytes = spin(c.consistency_max_prompt_bytes, 6000, REVIEW_PROMPT_BYTES_CEILING)
        self.c_mp_timeout = spin(c.mid_phase_timeout_seconds, 10, 3600)
        self.c_mp_budget = money(c.mid_phase_max_budget_usd)
        self.c_mp_prompt_bytes = spin(c.mid_phase_max_prompt_bytes, 4000, REVIEW_PROMPT_BYTES_CEILING)
        f.addRow("V4 consistency review:", self._row((("timeout s", self.c_cr_timeout), ("turns", self.c_cr_turns),
                                                       ("budget", self.c_cr_budget),
                                                       ("prompt bytes", self.c_cr_prompt_bytes))))
        f.addRow("Mid-phase diagnosis:", self._row((("timeout s", self.c_mp_timeout), ("budget", self.c_mp_budget),
                                                     ("prompt bytes", self.c_mp_prompt_bytes))))
        self.c_plan_budget = money(c.plan_max_budget_usd)
        self.c_review_budget = money(c.review_max_budget_usd)
        f.addRow("Max budget per call (--max-budget-usd):", pair(self.c_plan_budget, self.c_review_budget))
        self.c_retries = spin(c.auto_retries, 0, 1)
        f.addRow("Automatic retry of a failed Claude call:", self.c_retries)
        guard_note = QLabel("The installed Claude CLI has no --max-turns flag: turns and tool calls are counted "
                            "from its output and Claude is terminated when a limit is exceeded.")
        guard_note.setWordWrap(True)
        f.addRow(guard_note)
        self.c_read = QCheckBox("Allow the planner read-only repo tools (Read/Glob/Grep)")
        self.c_read.setChecked(s.claude.allow_repo_read_tools)
        f.addRow(self.c_read)
        self.c_args = _args_edit(s.claude.extra_args)
        f.addRow("Additional safe CLI args:", self.c_args)
        f.addRow(QLabel("Uses your existing Claude Code login. No API key is used or stored."))
        tabs.addTab(t, "Claude orchestrator")

        # ---------------- Worker
        t = QWidget()
        f = QFormLayout(t)
        self.w_exe = QLineEdit(s.worker.executable)
        f.addRow("Codex NInfer executable:", _path_row(self.w_exe, "Select codex-ninfer.exe", discover_codex_ninfer))
        warn = QLabel("⚠ Must be the dedicated codex-ninfer .exe. The normal `codex` launcher is never used and "
                      "there is no fallback.")
        warn.setWordWrap(True)
        f.addRow(warn)
        self.w_home = QLineEdit(s.worker.codex_home)
        f.addRow("CODEX_HOME (child process only):", self.w_home)
        self.w_env = QPlainTextEdit("\n".join(f"{k}={v}" for k, v in s.worker.extra_env.items()))
        self.w_env.setMaximumHeight(70)
        f.addRow("Extra child env (KEY=VALUE):", self.w_env)
        self.w_model = QLineEdit(s.worker.model)
        f.addRow("Worker model:", self.w_model)
        self.w_pass_model = QCheckBox("Pass --model to the worker (otherwise the profile's model is used)")
        self.w_pass_model.setChecked(s.worker.pass_model_flag)
        f.addRow(self.w_pass_model)
        self.w_sandbox = QComboBox()
        self.w_sandbox.addItems(["workspace-write", "read-only", "danger-full-access"])
        self.w_sandbox.setCurrentText(s.worker.sandbox if s.worker.sandbox in SANDBOX_MODES else "workspace-write")
        f.addRow("Sandbox (--sandbox):", self.w_sandbox)
        self.w_url = QLineEdit(s.worker.ninfer_base_url)
        f.addRow("NInfer base URL (status probe):", self.w_url)
        self.w_timeout = QSpinBox()
        self.w_timeout.setRange(60, 48 * 3600)
        self.w_timeout.setValue(s.worker.timeout_seconds)
        f.addRow("Timeout per attempt (s):", self.w_timeout)
        self.w_args = _args_edit(s.worker.extra_args)
        f.addRow("Additional safe exec args:", self.w_args)
        legacy_note = QLabel("These fields are the worker profile “NInfer Qwen (legacy settings)”. Sandbox and "
                             "timeout apply to every worker profile. More profiles: Worker Profiles tab.")
        legacy_note.setWordWrap(True)
        f.addRow(legacy_note)
        worker_tab = tabs.addTab(t, "Worker (Qwen / Codex NInfer)")

        # ---------------- Worker profiles (V5)
        self.profiles_tab = WorkerProfilesTab(s, in_use_provider, on_edit_legacy=lambda: tabs.setCurrentIndex(worker_tab),
                                              on_full_test=self._full_test_requested)
        tabs.addTab(self.profiles_tab, "Worker Profiles")

        # ---------------- Integrations
        t = QWidget()
        f = QFormLayout(t)
        self.i_cbm = QLineEdit(s.integrations.codebase_memory_executable)
        f.addRow("codebase-memory-mcp:", _path_row(self.i_cbm, "Select codebase-memory-mcp.exe", discover_codebase_memory))
        self.i_metrics_on = QCheckBox("Poll NInfer metrics endpoint")
        self.i_metrics_on.setChecked(s.integrations.metrics_enabled)
        f.addRow(self.i_metrics_on)
        self.i_metrics = QLineEdit(s.integrations.metrics_url)
        f.addRow("Metrics URL:", self.i_metrics)
        self.i_interval = QDoubleSpinBox()
        self.i_interval.setRange(0.5, 60)
        self.i_interval.setValue(s.integrations.metrics_interval_seconds)
        f.addRow("Metrics interval (s):", self.i_interval)
        self.i_unreal = QLineEdit(s.integrations.unreal_mcp_url)
        f.addRow("Unreal MCP URL (status only):", self.i_unreal)
        self.i_auto_index = QCheckBox("V2: index the repository in codebase-memory (exact root path) if it is not "
                                      "indexed yet")
        self.i_auto_index.setChecked(s.integrations.auto_index_codebase_memory)
        f.addRow(self.i_auto_index)
        self.i_index_timeout = QSpinBox()
        self.i_index_timeout.setRange(30, 24 * 3600)
        self.i_index_timeout.setValue(s.integrations.codebase_memory_index_timeout_seconds)
        f.addRow("Indexing timeout (s):", self.i_index_timeout)
        tabs.addTab(t, "Integrations")

        # ---------------- Orchestration
        t = QWidget()
        f = QFormLayout(t)
        self.o_orch = QComboBox()
        self.o_orch.addItems(["mock", "claude"])
        self.o_orch.setCurrentText(s.orchestration.orchestrator_mode)
        f.addRow("Orchestrator mode:", self.o_orch)
        self.o_worker = QComboBox()
        self.o_worker.addItems(["mock", "ninfer"])
        self.o_worker.setCurrentText(s.orchestration.worker_mode)
        f.addRow("Worker mode:", self.o_worker)
        note = QLabel("Real modes require the corresponding smoke test to have passed for the configured "
                      "executable. Real worker mode edits your repository.")
        note.setWordWrap(True)
        f.addRow(note)
        self.o_attempts = QSpinBox()
        self.o_attempts.setRange(1, 10)
        self.o_attempts.setValue(s.orchestration.max_attempts)
        f.addRow("Max attempts per task / phase:", self.o_attempts)
        self.o_pipeline = QComboBox()
        self.o_pipeline.addItem("V1 · task by task (Claude reviews every task)", "v1")
        self.o_pipeline.addItem("V2 · phases (one Qwen session per phase, Claude checkpoints)", "v2")
        self.o_pipeline.setCurrentIndex(max(0, self.o_pipeline.findData(s.orchestration.pipeline_version)))
        f.addRow("Pipeline for new plans:", self.o_pipeline)
        o = s.orchestration
        self.o_v2_turns = spin(o.v2_max_turns_per_attempt, 1, 20)
        self.o_v2_fix = spin(o.v2_local_fix_rounds, 0, 5)
        self.o_v2_session = spin(o.v2_max_session_turns, 1, 50)
        self.o_v2_replans = spin(o.v2_max_replans, 0, 10)
        v2_row = QWidget()
        vr = QHBoxLayout(v2_row)
        vr.setContentsMargins(0, 0, 0, 0)
        for label, w in (("turns/attempt", self.o_v2_turns), ("local fix rounds", self.o_v2_fix),
                         ("turns/session", self.o_v2_session), ("replans", self.o_v2_replans)):
            vr.addWidget(QLabel(label))
            vr.addWidget(w)
        vr.addStretch(1)
        f.addRow("V2 limits:", v2_row)
        # V2 REPAIRING guard (after a checkpoint RETRY); any limit stops the repair session.
        self.o_rep_wall = spin(o.repair_max_wall_seconds, 60, 24 * 3600)
        self.o_rep_tools = spin(o.repair_max_tool_calls, 1, 1000)
        self.o_rep_same = spin(o.repair_max_same_failure_rounds, 1, 20)
        self.o_rep_noimp = spin(o.repair_max_no_improvement_rounds, 1, 20)
        self.o_rep_grace = spin(o.repair_regression_grace_rounds, 0, 10)
        self.o_rep_probes = spin(o.repair_max_new_probe_files, 0, 50)
        self.o_rep_diag = spin(o.repair_max_diagnoses, 0, 10)
        rep_row = QWidget()
        rr = QHBoxLayout(rep_row)
        rr.setContentsMargins(0, 0, 0, 0)
        for label, w in (("wall s", self.o_rep_wall), ("tool calls", self.o_rep_tools),
                         ("same-failure rounds", self.o_rep_same),
                         ("no-improvement rounds", self.o_rep_noimp), ("regression grace", self.o_rep_grace),
                         ("probe files", self.o_rep_probes), ("diagnoses", self.o_rep_diag)):
            rr.addWidget(QLabel(label))
            rr.addWidget(w)
        rr.addStretch(1)
        f.addRow("V2 repair limits:", rep_row)
        self.o_human = QCheckBox("Allow Claude to request human verification")
        self.o_human.setToolTip("V2 phases whose verification contract opens a human gate: a checkpoint may pause the "
                                "run (AWAITING_HUMAN) and ask you to verify what automated evidence cannot settle.\n"
                                "When off, Claude decides PASS / RETRY / BLOCKED from the available evidence.")
        self.o_human.setChecked(o.human_verification_enabled)
        f.addRow(self.o_human)
        self.o_evidence = QCheckBox("Collect evidence for phases whose verification contract lists evidence "
                                    "requirements (V4)")
        self.o_evidence.setToolTip("Capability discovery, a Claude evidence plan over the discovered capabilities and "
                                   "bounded execution of that plan before the checkpoint. Verification only: evidence "
                                   "steps never edit, repair or commit anything.")
        self.o_evidence.setChecked(o.evidence_enabled)
        f.addRow(self.o_evidence)
        self.o_ev_wall = spin(o.evidence_max_wall_seconds, 10, 24 * 3600)
        self.o_ev_step = spin(o.evidence_step_timeout_seconds, 1, 24 * 3600)
        self.o_ev_steps = spin(o.evidence_max_steps, 0, 100)
        self.o_ev_output = spin(o.evidence_max_output_bytes, 256, 10_000_000)
        self.o_ev_artifacts = spin(o.evidence_max_artifacts, 0, 500)
        self.o_ev_retries = spin(o.evidence_max_retries, 0, 3)
        ev_row = QWidget()
        evl = QHBoxLayout(ev_row)
        evl.setContentsMargins(0, 0, 0, 0)
        for label, w in (("wall s", self.o_ev_wall), ("step timeout s", self.o_ev_step), ("steps", self.o_ev_steps),
                         ("output bytes", self.o_ev_output), ("artifacts", self.o_ev_artifacts),
                         ("retries", self.o_ev_retries)):
            evl.addWidget(QLabel(label))
            evl.addWidget(w)
        evl.addStretch(1)
        f.addRow("V4 evidence limits:", ev_row)
        self.o_ev_commands = _args_edit(o.evidence_commands)
        self.o_ev_commands.setPlaceholderText("one command line per line (offered to the evidence planner; "
                                              "destructive commands are never run)")
        f.addRow("V4 evidence commands:", self.o_ev_commands)
        self.o_consistency = QCheckBox("Review a checkpoint PASS that contradicts its own evidence judgment "
                                       "(bounded Claude consistency review)")
        self.o_consistency.setChecked(o.consistency_review_enabled)
        f.addRow(self.o_consistency)
        self.o_ev_replans = spin(o.evidence_max_replans, 0, 3)
        self.o_human_q = spin(o.human_max_questions, 1, 6)
        self.o_guard_files = spin(o.evidence_guard_max_files, 100, 200000)
        self.o_guard_mb = spin(max(1, o.evidence_guard_max_snapshot_bytes // 1_000_000), 0, 2000)
        f.addRow("V4 evidence hardening:", self._row((
            ("replans", self.o_ev_replans), ("human questions", self.o_human_q),
            ("non-git guard files", self.o_guard_files), ("snapshot MB", self.o_guard_mb))))
        self.o_art_cleanup = QCheckBox("clean up")
        self.o_art_cleanup.setChecked(o.evidence_artifact_cleanup_enabled)
        self.o_art_age = spin(o.evidence_artifact_max_age_days, 0, 3650)
        self.o_art_run_mb = spin(max(1, o.evidence_artifact_max_run_bytes // 1_000_000), 0, 100000)
        self.o_art_total_mb = spin(max(1, o.evidence_artifact_max_total_bytes // 1_000_000), 0, 1000000)
        f.addRow("Evidence artifacts:", self._row((
            ("", self.o_art_cleanup), ("max age days", self.o_art_age), ("MB per run", self.o_art_run_mb),
            ("MB total", self.o_art_total_mb))))
        self.o_providers = QPlainTextEdit("\n".join(f"{k}={v}" for k, v in o.evidence_provider_settings.items()))
        self.o_providers.setMaximumHeight(60)
        self.o_providers.setPlaceholderText("key=value per line, e.g. unreal.engine_root=C:\\Program Files\\Epic "
                                            "Games\\UE_5.4 · unreal.test_filter=MyGame · browser.enabled=true")
        f.addRow("Evidence provider settings:", self.o_providers)
        self.o_guard = QCheckBox("guard")
        self.o_guard.setToolTip("Stops a runaway implementation session (before its checkpoint) after the wall time, "
                                "tool calls or time without planned-subtask progress; Claude MID_PHASE_DIAGNOSE then "
                                "guides a fresh session. After the maximum diagnoses the phase blocks.")
        self.o_guard.setChecked(o.implementation_guard_enabled)
        self.o_impl_wall = spin(o.implementation_max_wall_seconds, 0, 24 * 3600)
        self.o_impl_tools = spin(o.implementation_max_tool_calls, 0, 10000)
        self.o_impl_noprog = spin(o.implementation_max_no_subtask_progress_seconds, 0, 24 * 3600)
        self.o_impl_diag = spin(o.implementation_max_diagnoses, 0, 10)
        f.addRow("Implementation guard:", self._row((
            ("", self.o_guard), ("wall s", self.o_impl_wall), ("tool calls", self.o_impl_tools),
            ("no subtask progress s", self.o_impl_noprog), ("diagnoses", self.o_impl_diag))))
        self.o_verify = QCheckBox("Run task verification commands (destructive ones always need approval)")
        self.o_verify.setChecked(s.orchestration.run_verification_commands)
        f.addRow(self.o_verify)
        self.o_vtimeout = QSpinBox()
        self.o_vtimeout.setRange(10, 24 * 3600)
        self.o_vtimeout.setValue(s.orchestration.verification_timeout_seconds)
        f.addRow("Verification timeout (s):", self.o_vtimeout)
        self.o_branch = QCheckBox("Create a dedicated working branch at START (real worker only)")
        self.o_branch.setChecked(s.orchestration.create_working_branch)
        f.addRow(self.o_branch)
        self.o_prefix = QLineEdit(s.orchestration.working_branch_prefix)
        f.addRow("Branch prefix:", self.o_prefix)
        self.o_precise = QCheckBox("Precise change attribution via temporary-index snapshots")
        self.o_precise.setChecked(s.orchestration.precise_git_snapshots)
        f.addRow(self.o_precise)
        f.addRow(QLabel("Automatic commits: OFF (not performed by this version)."))
        self.o_mock_retry = QLineEdit(", ".join(s.orchestration.mock_retry_task_ids))
        f.addRow("Mock: tasks to RETRY once:", self.o_mock_retry)
        tabs.addTab(t, "Orchestration")

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(buttons)

    def _full_test_requested(self, profile_id: str) -> None:
        """The full test runs a real (tiny) worker call on the SAVED profile: save first, then the main window
        starts it."""
        if QMessageBox.question(self, "Full test", f"Save the settings and run the full test of '{profile_id}' "
                                                   "now? It sends one tiny prompt through the worker and may use "
                                                   "paid API credits of the profile's provider.") != QMessageBox.Yes:
            return
        self.full_test_profile = profile_id
        self._save()
        if not self._accepted:  # the settings could not be saved: no test
            self.full_test_profile = None

    _accepted = False

    def accept(self) -> None:
        self._accepted = True
        super().accept()

    @staticmethod
    def _row(items: tuple[tuple[str, QWidget], ...]) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for label, widget in items:
            if label:
                h.addWidget(QLabel(label))
            h.addWidget(widget)
        h.addStretch(1)
        return w

    def _save(self) -> None:
        s = self.settings
        c_args = _parse_args(self.c_args)
        w_args = _parse_args(self.w_args)
        for a in c_args + w_args:
            if any(m in a.lower() for m in UNSAFE_ARG_MARKERS):
                QMessageBox.warning(self, "Settings", f"Refusing unsafe extra argument: {a}")
                return
        env: dict[str, str] = {}
        for line in self.w_env.toPlainText().splitlines():
            if not line.strip():
                continue
            if "=" not in line:
                QMessageBox.warning(self, "Settings", f"Invalid env line (expected KEY=VALUE): {line}")
                return
            k, v = line.split("=", 1)
            if k.strip().upper() == "CODEX_HOME":
                QMessageBox.warning(self, "Settings", "Set CODEX_HOME in its own field.")
                return
            env[k.strip()] = v.strip()

        w_exe = self.w_exe.text().strip()
        legacy_default = WorkerProfileRegistry(s).default_id() == LEGACY_PROFILE_ID
        if w_exe or (self.o_worker.currentText() == "ninfer" and legacy_default):
            try:
                validate_worker_executable(w_exe)
            except ExecutableRejected as exc:
                QMessageBox.critical(self, "Codex NInfer worker", f"{exc}\n\nThe worker will NOT fall back to "
                                                                 "any other executable.")
                return
        c_exe = self.c_exe.text().strip()
        if c_exe or self.o_orch.currentText() == "claude":
            try:
                validate_claude_executable(c_exe)
            except ExecutableRejected as exc:
                QMessageBox.critical(self, "Claude Code CLI", str(exc))
                return

        if self.w_sandbox.currentText() == "danger-full-access" and s.worker.sandbox != "danger-full-access":
            r = QMessageBox.question(
                self, "Worker sandbox",
                "danger-full-access runs the worker WITHOUT any sandbox: Qwen's commands can read and modify "
                "files outside the project repository.\n\nUse it anyway?")
            if r != QMessageBox.Yes:
                return
        w_home = self.w_home.text().strip()
        registry = WorkerProfileRegistry(s)
        default = registry.default_profile()
        if self.o_worker.currentText() == "ninfer" and registry.default_id() == LEGACY_PROFILE_ID \
                and not s.worker.verified_for(w_exe, w_home):
            QMessageBox.critical(
                self, "REAL worker mode",
                "REAL worker mode (ninfer) is only allowed after the worker smoke test has passed for\n\n"
                f"Worker executable:\n{w_exe}\n\nCODEX_HOME:\n{w_home}\n\n"
                "Save with worker mode 'mock', press 'Smoke test worker', then enable REAL mode.")
            return
        if self.o_worker.currentText() == "ninfer" and registry.default_id() != LEGACY_PROFILE_ID \
                and (default is None or not default.enabled or not default.is_verified()) \
                and self.full_test_profile != registry.default_id():
            QMessageBox.critical(
                self, "REAL worker mode",
                f"REAL worker mode needs a verified default worker profile. '{registry.default_id()}' has not passed "
                "its full test for its executable and CODEX_HOME.\n\nSave with worker mode 'mock', run the profile's "
                "Full test, then enable REAL mode.")
            return
        if c_exe != s.claude.executable:
            s.claude.verified_executable = ""
        s.claude.executable = c_exe
        s.claude.model = self.c_model.currentText().strip()
        s.claude.effort = self.c_effort.currentText().strip()
        s.claude.plan_timeout_seconds = self.c_plan_timeout.value()
        s.claude.review_timeout_seconds = self.c_review_timeout.value()
        s.claude.smoke_timeout_seconds = self.c_smoke_timeout.value()
        s.claude.plan_max_turns = self.c_plan_turns.value()
        s.claude.review_max_turns = self.c_review_turns.value()
        s.claude.plan_max_tool_calls = self.c_plan_tools.value()
        s.claude.review_model = self.c_review_model.currentText().strip()
        s.claude.review_max_prompt_bytes = self.c_review_prompt_bytes.value()
        s.claude.review_max_diff_bytes = min(self.c_review_diff_bytes.value(), self.c_review_prompt_bytes.value())
        s.claude.review_max_changed_files = self.c_review_files.value()
        s.claude.checkpoint_timeout_seconds = self.c_ck_timeout.value()
        s.claude.checkpoint_max_budget_usd = self.c_ck_budget.value()
        s.claude.checkpoint_max_prompt_bytes = self.c_ck_prompt_bytes.value()
        s.claude.checkpoint_max_diff_bytes = min(self.c_ck_diff_bytes.value(), self.c_ck_prompt_bytes.value())
        s.claude.checkpoint_max_changed_files = self.c_ck_files.value()
        s.claude.human_followup_timeout_seconds = self.c_hf_timeout.value()
        s.claude.human_followup_max_budget_usd = self.c_hf_budget.value()
        s.claude.human_followup_max_prompt_bytes = self.c_hf_prompt_bytes.value()
        s.claude.evidence_plan_timeout_seconds = self.c_ep_timeout.value()
        s.claude.evidence_plan_max_budget_usd = self.c_ep_budget.value()
        s.claude.evidence_plan_max_prompt_bytes = self.c_ep_prompt_bytes.value()
        s.claude.consistency_timeout_seconds = self.c_cr_timeout.value()
        s.claude.consistency_max_turns = self.c_cr_turns.value()
        s.claude.consistency_max_budget_usd = self.c_cr_budget.value()
        s.claude.consistency_max_prompt_bytes = self.c_cr_prompt_bytes.value()
        s.claude.mid_phase_timeout_seconds = self.c_mp_timeout.value()
        s.claude.mid_phase_max_budget_usd = self.c_mp_budget.value()
        s.claude.mid_phase_max_prompt_bytes = self.c_mp_prompt_bytes.value()
        s.claude.plan_max_budget_usd = self.c_plan_budget.value()
        s.claude.review_max_budget_usd = self.c_review_budget.value()
        s.claude.auto_retries = self.c_retries.value()
        s.claude.allow_repo_read_tools = self.c_read.isChecked()
        s.claude.extra_args = c_args
        s.worker.executable = w_exe
        s.worker.codex_home = w_home
        s.worker.extra_env = env
        s.worker.model = self.w_model.text().strip()
        s.worker.pass_model_flag = self.w_pass_model.isChecked()
        s.worker.sandbox = self.w_sandbox.currentText()
        s.worker.ninfer_base_url = self.w_url.text().strip()
        s.worker.timeout_seconds = self.w_timeout.value()
        s.worker.extra_args = w_args
        s.integrations.codebase_memory_executable = self.i_cbm.text().strip()
        s.integrations.metrics_enabled = self.i_metrics_on.isChecked()
        s.integrations.metrics_url = self.i_metrics.text().strip()
        s.integrations.metrics_interval_seconds = self.i_interval.value()
        s.integrations.unreal_mcp_url = self.i_unreal.text().strip()
        s.integrations.auto_index_codebase_memory = self.i_auto_index.isChecked()
        s.integrations.codebase_memory_index_timeout_seconds = self.i_index_timeout.value()
        s.orchestration.orchestrator_mode = self.o_orch.currentText()
        s.orchestration.worker_mode = self.o_worker.currentText()
        s.orchestration.max_attempts = self.o_attempts.value()
        s.orchestration.pipeline_version = self.o_pipeline.currentData() or "v1"
        s.orchestration.v2_max_turns_per_attempt = self.o_v2_turns.value()
        s.orchestration.v2_local_fix_rounds = self.o_v2_fix.value()
        s.orchestration.v2_max_session_turns = self.o_v2_session.value()
        s.orchestration.v2_max_replans = self.o_v2_replans.value()
        s.orchestration.repair_max_wall_seconds = self.o_rep_wall.value()
        s.orchestration.repair_max_tool_calls = self.o_rep_tools.value()
        s.orchestration.repair_max_same_failure_rounds = self.o_rep_same.value()
        s.orchestration.repair_max_no_improvement_rounds = self.o_rep_noimp.value()
        s.orchestration.repair_regression_grace_rounds = self.o_rep_grace.value()
        s.orchestration.repair_max_new_probe_files = self.o_rep_probes.value()
        s.orchestration.repair_max_diagnoses = self.o_rep_diag.value()
        s.orchestration.human_verification_enabled = self.o_human.isChecked()
        s.orchestration.evidence_enabled = self.o_evidence.isChecked()
        s.orchestration.evidence_max_wall_seconds = self.o_ev_wall.value()
        s.orchestration.evidence_step_timeout_seconds = self.o_ev_step.value()
        s.orchestration.evidence_max_steps = self.o_ev_steps.value()
        s.orchestration.evidence_max_output_bytes = self.o_ev_output.value()
        s.orchestration.evidence_max_artifacts = self.o_ev_artifacts.value()
        s.orchestration.evidence_max_retries = self.o_ev_retries.value()
        s.orchestration.evidence_commands = _parse_args(self.o_ev_commands)
        s.orchestration.consistency_review_enabled = self.o_consistency.isChecked()
        s.orchestration.evidence_max_replans = self.o_ev_replans.value()
        s.orchestration.human_max_questions = self.o_human_q.value()
        s.orchestration.evidence_guard_max_files = self.o_guard_files.value()
        s.orchestration.evidence_guard_max_snapshot_bytes = self.o_guard_mb.value() * 1_000_000
        s.orchestration.evidence_artifact_cleanup_enabled = self.o_art_cleanup.isChecked()
        s.orchestration.evidence_artifact_max_age_days = self.o_art_age.value()
        s.orchestration.evidence_artifact_max_run_bytes = self.o_art_run_mb.value() * 1_000_000
        s.orchestration.evidence_artifact_max_total_bytes = self.o_art_total_mb.value() * 1_000_000
        providers: dict[str, str] = {}
        for line in self.o_providers.toPlainText().splitlines():
            if not line.strip():
                continue
            if "=" not in line:
                QMessageBox.warning(self, "Settings", f"Invalid provider setting (expected key=value): {line}")
                return
            k, v = line.split("=", 1)
            providers[k.strip()] = v.strip()
        s.orchestration.evidence_provider_settings = providers
        s.orchestration.implementation_guard_enabled = self.o_guard.isChecked()
        s.orchestration.implementation_max_wall_seconds = self.o_impl_wall.value()
        s.orchestration.implementation_max_tool_calls = self.o_impl_tools.value()
        s.orchestration.implementation_max_no_subtask_progress_seconds = self.o_impl_noprog.value()
        s.orchestration.implementation_max_diagnoses = self.o_impl_diag.value()
        s.orchestration.run_verification_commands = self.o_verify.isChecked()
        s.orchestration.verification_timeout_seconds = self.o_vtimeout.value()
        s.orchestration.create_working_branch = self.o_branch.isChecked()
        s.orchestration.working_branch_prefix = self.o_prefix.text().strip() or "ai-orchestrator/run-"
        s.orchestration.precise_git_snapshots = self.o_precise.isChecked()
        s.orchestration.mock_retry_task_ids = [x.strip() for x in self.o_mock_retry.text().split(",") if x.strip()]
        self.accept()
