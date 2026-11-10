"""Headless GUI test: real widgets, real engine thread, mock providers."""
import os
import subprocess
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from app.config.settings import Settings, SettingsStore  # noqa: E402
from app.gui.bridge import EventBridge  # noqa: E402
from app.gui.main_window import MainWindow  # noqa: E402
from app.orchestration.orchestrator import EngineHost  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def pump(app, cond, timeout=30):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.01)
        if time.time() - t0 > timeout:
            raise AssertionError("condition not reached")


def test_gui_mock_run_end_to_end(qapp, tmp_path, monkeypatch):
    repo = tmp_path / "proj"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    s = Settings()
    s.integrations.metrics_enabled = False
    s.orchestration.mock_step_delay = 0.01
    s.last_repo_path = str(repo)
    store = SettingsStore(tmp_path / "settings.json")
    bridge = EventBridge()
    host = EngineHost(s, bridge.emit, str(tmp_path / "gui.db"))
    win = MainWindow(s, store, bridge, host)
    host.start()
    win.engine_started()
    win.show()
    try:
        # Error from the engine (planning without a goal) is shown, not crashing.
        monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
        win.task_input.setPlainText("Implement an inventory system")
        win.btn_plan.click()
        # the "state" event precedes the "plan" event; wait for both
        pump(qapp, lambda: win.state == "PLAN_READY" and win.tasks.topLevelItemCount() == 4)
        assert win.btn_start.isEnabled() and not win.btn_pause.isEnabled()
        assert not win.attempts, "PLAN must not start any worker attempt"
        win.btn_start.click()
        pump(qapp, lambda: win.state == "COMPLETED")
        statuses = [win.tasks.topLevelItem(i).text(2) for i in range(4)]
        assert statuses == ["PASSED"] * 4
        assert len(win.attempts["T002"]) == 2  # mock RETRY then PASS
        win.tasks.setCurrentItem(win.tasks.topLevelItem(1))
        qapp.processEvents()
        html = win.details.toPlainText()
        assert "RETRY" in html and "PASS" in html
        pump(qapp, lambda: win.project.history.count() >= 1)
        win.grab().save(str(tmp_path / "screenshot.png"))
    finally:
        win.close()


def test_smoke_result_from_engine_thread_reaches_gui(qapp, tmp_path, monkeypatch):
    """Regression: the smoke result is produced on the engine thread and must be
    shown/persisted on the GUI thread."""
    import threading
    s = Settings()
    s.integrations.metrics_enabled = False
    store = SettingsStore(tmp_path / "settings.json")
    bridge = EventBridge()
    host = EngineHost(s, bridge.emit, str(tmp_path / "gui2.db"))
    win = MainWindow(s, store, bridge, host)
    host.start()
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda p, title, text: shown.append((title, text)))
    payload = {"target": "worker", "ok": True, "executable": r"C:\x\codex-ninfer.exe", "detail": "ok",
               "extra": {"codex_home": r"C:\u\.codex-ninfer", "reply": "QWEN_WORKER_OK", "exit_code": 0,
                         "elapsed": 1.0, "files_created": []},
               "verified_at": 1.0, "verified_executable": r"C:\x\codex-ninfer.exe",
               "verified_codex_home": r"C:\u\.codex-ninfer"}
    try:
        threading.Thread(target=lambda: bridge.emit("smoke", payload)).start()
        pump(qapp, lambda: shown)
        title, text = shown[0]
        assert "PASSED" in title
        for part in ["Worker executable:\n" + r"C:\x\codex-ninfer.exe", "CODEX_HOME:\n" + r"C:\u\.codex-ninfer",
                     "Response:\nQWEN_WORKER_OK"]:
            assert part in text
        saved = store.load().worker
        assert saved.verified_executable == r"C:\x\codex-ninfer.exe"
        assert saved.verified_codex_home == r"C:\u\.codex-ninfer"
    finally:
        win.close()


def test_task_list_and_footer_render_subtask_progress_from_the_worker_todo(qapp):
    """V2.1.1: a todo update that completes S1 reaches the tree (☑ S1) and the footer (subtasks 1/3)."""
    from types import SimpleNamespace
    from app.gui.task_list import TaskList
    from app.models.phase import parse_phases
    ph = parse_phases([{"id": "P1", "title": "Chase", "description": "d", "acceptance_criteria": ["ok"],
                        "subtasks": [{"id": "S1", "title": "Fix chase"}, {"id": "S2", "title": "Search/return"},
                                     {"id": "S3", "title": "Pedestrians"}]}])[0]
    tree = TaskList()
    tree.set_tasks([ph.to_dict()])
    footer = lambda: MainWindow._phase_progress_text(SimpleNamespace(tasks=tree), "P1")  # noqa: E731
    assert "subtasks 0/3" in footer()
    ph.todo = [{"text": "Explore repo", "completed": True}, {"text": "S1: Fix chase give-up", "completed": True},
               {"text": "S2: Search/return", "completed": False}, {"text": "S3: Pedestrians", "completed": False},
               {"text": "Verify", "completed": False}]
    assert ph.apply_todo(ph.todo) == ["S1"]
    tree.update_task(ph.to_dict())  # what the "task" event does
    item = tree.topLevelItem(0)
    assert [item.child(i).text(0).strip() for i in range(item.childCount())] == ["☑ S1", "☐ S2", "☐ S3"]
    assert "subtasks 1/3" in footer() and "todo 2/5" in footer()
